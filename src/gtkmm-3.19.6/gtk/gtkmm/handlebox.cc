// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!

#undef GTK_DISABLE_DEPRECATED
#define GDK_DISABLE_DEPRECATION_WARNINGS 1
 

#ifndef GTKMM_DISABLE_DEPRECATED


#include <glibmm.h>

#include <gtkmm/handlebox.h>
#include <gtkmm/private/handlebox_p.h>


/*
 * Copyright 1998-2002 The gtkmm Development Team
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

#include <gtk/gtk.h>
#include <gdkmm/window.h>

namespace Gtk
{

} //namespace Gtk

namespace
{


static void HandleBox_signal_child_attached_callback(GtkHandleBox* self, GtkWidget* p0,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,Widget* > SlotType;

  auto obj = dynamic_cast<HandleBox*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo HandleBox_signal_child_attached_info =
{
  "child_attached",
  (GCallback) &HandleBox_signal_child_attached_callback,
  (GCallback) &HandleBox_signal_child_attached_callback
};


static void HandleBox_signal_child_detached_callback(GtkHandleBox* self, GtkWidget* p0,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,Widget* > SlotType;

  auto obj = dynamic_cast<HandleBox*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo HandleBox_signal_child_detached_info =
{
  "child_detached",
  (GCallback) &HandleBox_signal_child_detached_callback,
  (GCallback) &HandleBox_signal_child_detached_callback
};


} // anonymous namespace


namespace Glib
{

Gtk::HandleBox* wrap(GtkHandleBox* object, bool take_copy)
{
  return dynamic_cast<Gtk::HandleBox *> (Glib::wrap_auto ((GObject*)(object), take_copy));
}

} /* namespace Glib */

namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& HandleBox_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &HandleBox_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_handle_box_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:

  }

  return *this;
}


void HandleBox_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


  klass->child_attached = &child_attached_callback;
  klass->child_detached = &child_detached_callback;
}


void HandleBox_Class::child_attached_callback(GtkHandleBox* self, GtkWidget* p0)
{
  const auto obj_base = static_cast<Glib::ObjectBase*>(
      Glib::ObjectBase::_get_current_wrapper((GObject*)self));

  // Non-gtkmmproc-generated custom classes implicitly call the default
  // Glib::ObjectBase constructor, which sets is_derived_. But gtkmmproc-
  // generated classes can use this optimisation, which avoids the unnecessary
  // parameter conversions if there is no possibility of the virtual function
  // being overridden:
  if(obj_base && obj_base->is_derived_())
  {
    const auto obj = dynamic_cast<CppObjectType* const>(obj_base);
    if(obj) // This can be NULL during destruction.
    {
      try // Trap C++ exceptions which would normally be lost because this is a C callback.
      {
        // Call the virtual member method, which derived classes might override.
        obj->on_child_attached(Glib::wrap(p0)
);
        return;
      }
      catch(...)
      {
        Glib::exception_handlers_invoke();
      }
    }
  }

  const auto base = static_cast<BaseClassType*>(
        g_type_class_peek_parent(G_OBJECT_GET_CLASS(self)) // Get the parent class of the object class (The original underlying C class).
    );

  // Call the original underlying C function:
  if(base && base->child_attached)
    (*base->child_attached)(self, p0);
}
void HandleBox_Class::child_detached_callback(GtkHandleBox* self, GtkWidget* p0)
{
  const auto obj_base = static_cast<Glib::ObjectBase*>(
      Glib::ObjectBase::_get_current_wrapper((GObject*)self));

  // Non-gtkmmproc-generated custom classes implicitly call the default
  // Glib::ObjectBase constructor, which sets is_derived_. But gtkmmproc-
  // generated classes can use this optimisation, which avoids the unnecessary
  // parameter conversions if there is no possibility of the virtual function
  // being overridden:
  if(obj_base && obj_base->is_derived_())
  {
    const auto obj = dynamic_cast<CppObjectType* const>(obj_base);
    if(obj) // This can be NULL during destruction.
    {
      try // Trap C++ exceptions which would normally be lost because this is a C callback.
      {
        // Call the virtual member method, which derived classes might override.
        obj->on_child_detached(Glib::wrap(p0)
);
        return;
      }
      catch(...)
      {
        Glib::exception_handlers_invoke();
      }
    }
  }

  const auto base = static_cast<BaseClassType*>(
        g_type_class_peek_parent(G_OBJECT_GET_CLASS(self)) // Get the parent class of the object class (The original underlying C class).
    );

  // Call the original underlying C function:
  if(base && base->child_detached)
    (*base->child_detached)(self, p0);
}


Glib::ObjectBase* HandleBox_Class::wrap_new(GObject* o)
{
  return manage(new HandleBox((GtkHandleBox*)(o)));

}


/* The implementation: */

HandleBox::HandleBox(const Glib::ConstructParams& construct_params)
:
  Gtk::Bin(construct_params)
{
  }

HandleBox::HandleBox(GtkHandleBox* castitem)
:
  Gtk::Bin((GtkBin*)(castitem))
{
  }


HandleBox::HandleBox(HandleBox&& src) noexcept
: Gtk::Bin(std::move(src))
{}

HandleBox& HandleBox::operator=(HandleBox&& src) noexcept
{
  Gtk::Bin::operator=(std::move(src));
  return *this;
}

HandleBox::~HandleBox() noexcept
{
  destroy_();
}

HandleBox::CppClassType HandleBox::handlebox_class_; // initialize static member

GType HandleBox::get_type()
{
  return handlebox_class_.init().get_type();
}


GType HandleBox::get_base_type()
{
  return gtk_handle_box_get_type();
}


HandleBox::HandleBox()
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::Bin(Glib::ConstructParams(handlebox_class_.init()))
{
  

}

void HandleBox::set_shadow_type(ShadowType type)
{
  gtk_handle_box_set_shadow_type(gobj(), ((GtkShadowType)(type)));
}

ShadowType HandleBox::get_shadow_type() const
{
  return ((ShadowType)(gtk_handle_box_get_shadow_type(const_cast<GtkHandleBox*>(gobj()))));
}

void HandleBox::set_handle_position(PositionType position)
{
  gtk_handle_box_set_handle_position(gobj(), ((GtkPositionType)(position)));
}

PositionType HandleBox::get_handle_position() const
{
  return ((PositionType)(gtk_handle_box_get_handle_position(const_cast<GtkHandleBox*>(gobj()))));
}

void HandleBox::set_snap_edge(PositionType edge)
{
  gtk_handle_box_set_snap_edge(gobj(), ((GtkPositionType)(edge)));
}

PositionType HandleBox::get_snap_edge() const
{
  return ((PositionType)(gtk_handle_box_get_snap_edge(const_cast<GtkHandleBox*>(gobj()))));
}

bool HandleBox::is_child_detached() const
{
  return gtk_handle_box_get_child_detached(const_cast<GtkHandleBox*>(gobj()));
}


Glib::SignalProxy1< void,Widget* > HandleBox::signal_child_attached()
{
  return Glib::SignalProxy1< void,Widget* >(this, &HandleBox_signal_child_attached_info);
}


Glib::SignalProxy1< void,Widget* > HandleBox::signal_child_detached()
{
  return Glib::SignalProxy1< void,Widget* >(this, &HandleBox_signal_child_detached_info);
}


Glib::PropertyProxy< ShadowType > HandleBox::property_shadow_type() 
{
  return Glib::PropertyProxy< ShadowType >(this, "shadow-type");
}

Glib::PropertyProxy_ReadOnly< ShadowType > HandleBox::property_shadow_type() const
{
  return Glib::PropertyProxy_ReadOnly< ShadowType >(this, "shadow-type");
}

Glib::PropertyProxy< PositionType > HandleBox::property_handle_position() 
{
  return Glib::PropertyProxy< PositionType >(this, "handle-position");
}

Glib::PropertyProxy_ReadOnly< PositionType > HandleBox::property_handle_position() const
{
  return Glib::PropertyProxy_ReadOnly< PositionType >(this, "handle-position");
}

Glib::PropertyProxy< PositionType > HandleBox::property_snap_edge() 
{
  return Glib::PropertyProxy< PositionType >(this, "snap-edge");
}

Glib::PropertyProxy_ReadOnly< PositionType > HandleBox::property_snap_edge() const
{
  return Glib::PropertyProxy_ReadOnly< PositionType >(this, "snap-edge");
}

Glib::PropertyProxy< bool > HandleBox::property_snap_edge_set() 
{
  return Glib::PropertyProxy< bool >(this, "snap-edge-set");
}

Glib::PropertyProxy_ReadOnly< bool > HandleBox::property_snap_edge_set() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "snap-edge-set");
}

Glib::PropertyProxy_ReadOnly< bool > HandleBox::property_child_detached() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "child-detached");
}


void Gtk::HandleBox::on_child_attached(Widget* child)
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->child_attached)
    (*base->child_attached)(gobj(),(GtkWidget*)Glib::unwrap(child));
}
void Gtk::HandleBox::on_child_detached(Widget* child)
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->child_detached)
    (*base->child_detached)(gobj(),(GtkWidget*)Glib::unwrap(child));
}


} // namespace Gtk

#endif // GTKMM_DISABLE_DEPRECATED


