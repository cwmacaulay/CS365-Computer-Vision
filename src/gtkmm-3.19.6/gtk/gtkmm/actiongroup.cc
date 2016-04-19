// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!

#undef GTK_DISABLE_DEPRECATED
#define GDK_DISABLE_DEPRECATION_WARNINGS 1
 

#include <glibmm.h>

#include <gtkmm/actiongroup.h>
#include <gtkmm/private/actiongroup_p.h>


/* Copyright 2003 The gtkmm Development Team
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <glibmm/vectorutils.h>

#include <gtk/gtk.h>
#include <gtkmm/accelmap.h>


namespace Gtk
{

void ActionGroup::add(const Glib::RefPtr<Action>& action)
{
  gtk_action_group_add_action_with_accel(gobj(), Glib::unwrap(action),
    nullptr /* accelerator=nullptr means use the stock accelerator if this is a stock item */ );
}

void ActionGroup::add(const Glib::RefPtr<Action>& action, const AccelKey& accel_key)
{
  // We need to half-duplicate the gtk_action_group_add_action_with_accel() implementation, because we want to
  // use AccelKey, not just the accelerator string format that is _one_ of the ways to create an AccelKey. murrayc.

  //The AccelKey might have been constructed from key+mod or from an accelerator string,
  //but it the end that just produces a key+mod.
  guint key = accel_key.get_key();
  auto mod = accel_key.get_mod();
  if(key)
  {
    // The AccelKey constructor can take an accel path, so I suppose we should not ignore it,
    // but I can't imagine how it would be useful with the UIManager. murrayc.
    auto accel_path = accel_key.get_path();
    if(accel_path.empty())
    {
      //Copied from the gtk_action_group_add_action_with_accel() implementation:

      gchar* action_name = nullptr;
      g_object_get(G_OBJECT(Glib::unwrap(action)), "name", &action_name, nullptr);
      if(action_name)
      {
        accel_path =  "<Actions>/" + get_name() + '/' + action_name;
        g_free(action_name);
      }
    }

    AccelMap::add_entry(accel_path, key, mod);

    action->set_accel_path(accel_path);
  }

  gtk_action_group_add_action(gobj(), action->gobj());
}


void ActionGroup::add(const Glib::RefPtr<Action>& action, const Action::SlotActivate& slot)
{
  //This is meant to be similar to the gtk_action_group_add_actions() convenience method that is used from C.
  //This also does the work of gtk_action_group_add_toggle_actions() and gtk_action_group_add_radio_actions_full(),
  //because the extra stuff there is already done when the Action (or a derived type, such as ToggleAction) is create()ed.

  action->signal_activate().connect(slot);

  //We probably don't need to use translate_string(), because the label and tooltip will be regular translated strings in our C++ application code.
  //But C needs it because gtk_action_group_add_actions() takes a static array of GtkActionEntrys  whose fields
  //can not be translated with gettext macros.
  //But we should soon know if menus and toolbars are not translated in non-english locales. murrayc.

  add(action);
}

void ActionGroup::add(const Glib::RefPtr<Action>& action, const AccelKey& accel_key, const Action::SlotActivate& slot)
{
  //This is meant to be similar to the gtk_action_group_add_actions() convenience method that is used from C.
  //This also does the work of gtk_action_group_add_toggle_actions() and gtk_action_group_add_radio_actions_full(),
  //because the extra stuff there is already done when the Action (or a derived type, such as ToggleAction) is create()ed.

  action->signal_activate().connect(slot);
  add(action, accel_key);

  /*
  //Create the accel path (a kind of unique key):
  Glib::ustring accel_path = "<Actions>/" + get_name() + "/" + action->get_name();

  //Register the accel path:
  Gtk::AccelMap::add_entry(accel_path, accel_key.get_key(), accel_key.get_mod());

  //USe the accel path:
  action->set_accel_path(accel_path);
  */
}


} // namespace Gtk


namespace
{


static void ActionGroup_signal_connect_proxy_callback(GtkActionGroup* self, GtkAction* p0,GtkWidget* p1,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,const Glib::RefPtr<Action>&,Widget* > SlotType;

  auto obj = dynamic_cast<ActionGroup*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0, true)
, Glib::wrap(p1)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo ActionGroup_signal_connect_proxy_info =
{
  "connect_proxy",
  (GCallback) &ActionGroup_signal_connect_proxy_callback,
  (GCallback) &ActionGroup_signal_connect_proxy_callback
};


static void ActionGroup_signal_disconnect_proxy_callback(GtkActionGroup* self, GtkAction* p0,GtkWidget* p1,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,const Glib::RefPtr<Action>&,Widget* > SlotType;

  auto obj = dynamic_cast<ActionGroup*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0, true)
, Glib::wrap(p1)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo ActionGroup_signal_disconnect_proxy_info =
{
  "disconnect_proxy",
  (GCallback) &ActionGroup_signal_disconnect_proxy_callback,
  (GCallback) &ActionGroup_signal_disconnect_proxy_callback
};


static void ActionGroup_signal_pre_activate_callback(GtkActionGroup* self, GtkAction* p0,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,const Glib::RefPtr<Action>& > SlotType;

  auto obj = dynamic_cast<ActionGroup*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0, true)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo ActionGroup_signal_pre_activate_info =
{
  "pre_activate",
  (GCallback) &ActionGroup_signal_pre_activate_callback,
  (GCallback) &ActionGroup_signal_pre_activate_callback
};


static void ActionGroup_signal_post_activate_callback(GtkActionGroup* self, GtkAction* p0,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,const Glib::RefPtr<Action>& > SlotType;

  auto obj = dynamic_cast<ActionGroup*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0, true)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo ActionGroup_signal_post_activate_info =
{
  "post_activate",
  (GCallback) &ActionGroup_signal_post_activate_callback,
  (GCallback) &ActionGroup_signal_post_activate_callback
};


} // anonymous namespace


namespace Glib
{

Glib::RefPtr<Gtk::ActionGroup> wrap(GtkActionGroup* object, bool take_copy)
{
  return Glib::RefPtr<Gtk::ActionGroup>( dynamic_cast<Gtk::ActionGroup*> (Glib::wrap_auto ((GObject*)(object), take_copy)) );
  //We use dynamic_cast<> in case of multiple inheritance.
}

} /* namespace Glib */


namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& ActionGroup_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &ActionGroup_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_action_group_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:
  Buildable::add_interface(get_type());

  }

  return *this;
}


void ActionGroup_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


}


Glib::ObjectBase* ActionGroup_Class::wrap_new(GObject* object)
{
  return new ActionGroup((GtkActionGroup*)object);
}


/* The implementation: */

GtkActionGroup* ActionGroup::gobj_copy()
{
  reference();
  return gobj();
}

ActionGroup::ActionGroup(const Glib::ConstructParams& construct_params)
:
  Glib::Object(construct_params)
{

}

ActionGroup::ActionGroup(GtkActionGroup* castitem)
:
  Glib::Object((GObject*)(castitem))
{}


ActionGroup::ActionGroup(ActionGroup&& src) noexcept
: Glib::Object(std::move(src))
  , Buildable(std::move(src))
{}

ActionGroup& ActionGroup::operator=(ActionGroup&& src) noexcept
{
  Glib::Object::operator=(std::move(src));
  Buildable::operator=(std::move(src));
  return *this;
}

ActionGroup::~ActionGroup() noexcept
{}


ActionGroup::CppClassType ActionGroup::actiongroup_class_; // initialize static member

GType ActionGroup::get_type()
{
  return actiongroup_class_.init().get_type();
}


GType ActionGroup::get_base_type()
{
  return gtk_action_group_get_type();
}


ActionGroup::ActionGroup(const Glib::ustring& name)
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Glib::Object(Glib::ConstructParams(actiongroup_class_.init(), "name", name.c_str(), nullptr))
{
  

}

Glib::RefPtr<ActionGroup> ActionGroup::create(const Glib::ustring& name)
{
  return Glib::RefPtr<ActionGroup>( new ActionGroup(name) );
}

Glib::ustring ActionGroup::get_name() const
{
  return Glib::convert_const_gchar_ptr_to_ustring(gtk_action_group_get_name(const_cast<GtkActionGroup*>(gobj())));
}

bool ActionGroup::get_sensitive() const
{
  return gtk_action_group_get_sensitive(const_cast<GtkActionGroup*>(gobj()));
}

void ActionGroup::set_sensitive(bool sensitive)
{
  gtk_action_group_set_sensitive(gobj(), static_cast<int>(sensitive));
}

bool ActionGroup::get_visible() const
{
  return gtk_action_group_get_visible(const_cast<GtkActionGroup*>(gobj()));
}

void ActionGroup::set_visible(bool visible)
{
  gtk_action_group_set_visible(gobj(), static_cast<int>(visible));
}

Glib::RefPtr<AccelGroup> ActionGroup::get_accel_group()
{
  Glib::RefPtr<AccelGroup> retvalue = Glib::wrap(gtk_action_group_get_accel_group(gobj()));
  if(retvalue)
    retvalue->reference(); //The function does not do a ref for us.
  return retvalue;
}

Glib::RefPtr<const AccelGroup> ActionGroup::get_accel_group() const
{
  return const_cast<ActionGroup*>(this)->get_accel_group();
}

void ActionGroup::set_accel_group(const Glib::RefPtr<AccelGroup>& accel_group)
{
  gtk_action_group_set_accel_group(gobj(), Glib::unwrap(accel_group));
}

Glib::RefPtr<Action> ActionGroup::get_action(const Glib::ustring& action_name)
{
  Glib::RefPtr<Action> retvalue = Glib::wrap(gtk_action_group_get_action(gobj(), action_name.c_str()));
  if(retvalue)
    retvalue->reference(); //The function does not do a ref for us.
  return retvalue;
}

Glib::RefPtr<const Action> ActionGroup::get_action(const Glib::ustring& action_name) const
{
  return const_cast<ActionGroup*>(this)->get_action(action_name);
}

std::vector< Glib::RefPtr<Action> > ActionGroup::get_actions()
{
  return Glib::ListHandler< Glib::RefPtr<Action> >::list_to_vector(gtk_action_group_list_actions(gobj()), Glib::OWNERSHIP_SHALLOW);
}

std::vector< Glib::RefPtr<const Action> > ActionGroup::get_actions() const
{
  return Glib::ListHandler< Glib::RefPtr<const Action> >::list_to_vector(gtk_action_group_list_actions(const_cast<GtkActionGroup*>(gobj())), Glib::OWNERSHIP_SHALLOW);
}

void ActionGroup::remove(const Glib::RefPtr<Action>& action)
{
  gtk_action_group_remove_action(gobj(), Glib::unwrap(action));
}

Glib::ustring ActionGroup::translate_string(const Glib::ustring& string) const
{
  return Glib::convert_const_gchar_ptr_to_ustring(gtk_action_group_translate_string(const_cast<GtkActionGroup*>(gobj()), string.c_str()));
}


Glib::SignalProxy2< void,const Glib::RefPtr<Action>&,Widget* > ActionGroup::signal_connect_proxy()
{
  return Glib::SignalProxy2< void,const Glib::RefPtr<Action>&,Widget* >(this, &ActionGroup_signal_connect_proxy_info);
}


Glib::SignalProxy2< void,const Glib::RefPtr<Action>&,Widget* > ActionGroup::signal_disconnect_proxy()
{
  return Glib::SignalProxy2< void,const Glib::RefPtr<Action>&,Widget* >(this, &ActionGroup_signal_disconnect_proxy_info);
}


Glib::SignalProxy1< void,const Glib::RefPtr<Action>& > ActionGroup::signal_pre_activate()
{
  return Glib::SignalProxy1< void,const Glib::RefPtr<Action>& >(this, &ActionGroup_signal_pre_activate_info);
}


Glib::SignalProxy1< void,const Glib::RefPtr<Action>& > ActionGroup::signal_post_activate()
{
  return Glib::SignalProxy1< void,const Glib::RefPtr<Action>& >(this, &ActionGroup_signal_post_activate_info);
}


Glib::PropertyProxy_ReadOnly< Glib::ustring > ActionGroup::property_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "name");
}

Glib::PropertyProxy< bool > ActionGroup::property_sensitive() 
{
  return Glib::PropertyProxy< bool >(this, "sensitive");
}

Glib::PropertyProxy_ReadOnly< bool > ActionGroup::property_sensitive() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "sensitive");
}

Glib::PropertyProxy< bool > ActionGroup::property_visible() 
{
  return Glib::PropertyProxy< bool >(this, "visible");
}

Glib::PropertyProxy_ReadOnly< bool > ActionGroup::property_visible() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "visible");
}

Glib::PropertyProxy< Glib::RefPtr<AccelGroup> > ActionGroup::property_accel_group() 
{
  return Glib::PropertyProxy< Glib::RefPtr<AccelGroup> >(this, "accel-group");
}

Glib::PropertyProxy_ReadOnly< Glib::RefPtr<AccelGroup> > ActionGroup::property_accel_group() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::RefPtr<AccelGroup> >(this, "accel-group");
}


} // namespace Gtk


