// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!


#include <glibmm.h>

#include <gtkmm/cellarea.h>
#include <gtkmm/private/cellarea_p.h>


/* Copyright 2010 The gtkmm Development Team
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


namespace //anonymous namespace
{

static gboolean proxy_foreach_callback(GtkCellRenderer* cell, void* data)
{
  typedef Gtk::CellArea::SlotForeach SlotType;
  auto& slot = *static_cast<SlotType*>(data);

  try
  {
    return slot(Glib::wrap(cell));
  }
  catch(...)
  {
    Glib::exception_handlers_invoke();
  }

  return FALSE;
}

static gboolean proxy_foreach_alloc_callback(GtkCellRenderer* cell, const GdkRectangle* cell_area, const GdkRectangle* cell_background, void* data)
{
  typedef Gtk::CellArea::SlotForeachAlloc SlotType;
  auto& slot = *static_cast<SlotType*>(data);

  try
  {
    return slot(Glib::wrap(cell), Glib::wrap(cell_area), Glib::wrap(cell_background));
  }
  catch(...)
  {
    Glib::exception_handlers_invoke();
  }

  return FALSE;
}


} //anonymous namespace


namespace Gtk
{

void CellArea::foreach(const SlotForeach& slot)
{
  SlotForeach slot_copy(slot);
  gtk_cell_area_foreach(const_cast<GtkCellArea*>(gobj()), &proxy_foreach_callback, &slot_copy);
}

void CellArea::foreach(const Glib::RefPtr<CellAreaContext>& context, Widget* widget, const Gdk::Rectangle& cell_area, const Gdk::Rectangle& background_area, const SlotForeachAlloc& slot)
{
  SlotForeachAlloc slot_copy(slot);
  gtk_cell_area_foreach_alloc(const_cast<GtkCellArea*>(gobj()), Glib::unwrap(context), Glib::unwrap(widget), cell_area.gobj(), background_area.gobj(), &proxy_foreach_alloc_callback, &slot_copy);
}


//These vfunc callbacks are custom implemented because we want the output
//arguments of the C++ vfuncs to be int& (not int*), and the vfunc_callback
//functions may be called from gtk+ with a NULL pointer.
void CellArea_Class::get_preferred_width_vfunc_callback(GtkCellArea* self, GtkCellAreaContext* context, GtkWidget* widget, gint* minimum_width, gint* natural_width)
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
        int min_width = 0;
        int nat_width = 0;
        obj->get_preferred_width_vfunc(Glib::wrap(context, true),
             *Glib::wrap(widget),
             (minimum_width ? *minimum_width : min_width),
             (natural_width ? *natural_width : nat_width));
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
  if(base && base->get_preferred_width)
    (*base->get_preferred_width)(self, context, widget, minimum_width, natural_width);
}

void CellArea_Class::get_preferred_height_for_width_vfunc_callback(GtkCellArea* self, GtkCellAreaContext* context, GtkWidget* widget, gint width, gint* minimum_height, gint* natural_height)
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
        int min_height = 0;
        int nat_height = 0;
        obj->get_preferred_height_for_width_vfunc(Glib::wrap(context, true),
             *Glib::wrap(widget), width,
             (minimum_height ? *minimum_height : min_height),
             (natural_height ? *natural_height : nat_height));
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
  if(base && base->get_preferred_height_for_width)
    (*base->get_preferred_height_for_width)(self, context, widget, width, minimum_height, natural_height);
}

void CellArea_Class::get_preferred_height_vfunc_callback(GtkCellArea* self, GtkCellAreaContext* context, GtkWidget* widget, gint* minimum_height, gint* natural_height)
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
        int min_height = 0;
        int nat_height = 0;
        obj->get_preferred_height_vfunc(Glib::wrap(context, true),
             *Glib::wrap(widget),
             (minimum_height ? *minimum_height : min_height),
             (natural_height ? *natural_height : nat_height));
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
  if(base && base->get_preferred_height)
    (*base->get_preferred_height)(self, context, widget, minimum_height, natural_height);
}

void CellArea_Class::get_preferred_width_for_height_vfunc_callback(GtkCellArea* self, GtkCellAreaContext* context, GtkWidget* widget, gint height, gint* minimum_width, gint* natural_width)
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
        int min_width = 0;
        int nat_width = 0;
        obj->get_preferred_width_for_height_vfunc(Glib::wrap(context, true),
             *Glib::wrap(widget), height,
             (minimum_width ? *minimum_width : min_width),
             (natural_width ? *natural_width : nat_width));
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
  if(base && base->get_preferred_width_for_height)
    (*base->get_preferred_width_for_height)(self, context, widget, height, minimum_width, natural_width);
}


} //namespace Gtk


namespace
{


static void CellArea_signal_apply_attributes_callback(GtkCellArea* self, GtkTreeModel* p0,GtkTreeIter* p1,gboolean p2,gboolean p3,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,const Glib::RefPtr<TreeModel>&,const TreeModel::iterator&,bool,bool > SlotType;

  auto obj = dynamic_cast<CellArea*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0, true)
, Gtk::TreeModel::iterator(p0, p1)
, p2
, p3
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo CellArea_signal_apply_attributes_info =
{
  "apply-attributes",
  (GCallback) &CellArea_signal_apply_attributes_callback,
  (GCallback) &CellArea_signal_apply_attributes_callback
};


static void CellArea_signal_add_editable_callback(GtkCellArea* self, GtkCellRenderer* p0,GtkCellEditable* p1,GdkRectangle* p2,const gchar* p3,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,CellRenderer*,CellEditable*,const Gdk::Rectangle&,const Glib::ustring& > SlotType;

  auto obj = dynamic_cast<CellArea*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0)
, dynamic_cast<CellEditable*>(Glib::wrap_auto((GObject*)(p1), false))
, Glib::wrap(p2)
, Glib::convert_const_gchar_ptr_to_ustring(p3)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo CellArea_signal_add_editable_info =
{
  "add-editable",
  (GCallback) &CellArea_signal_add_editable_callback,
  (GCallback) &CellArea_signal_add_editable_callback
};


static void CellArea_signal_remove_editable_callback(GtkCellArea* self, GtkCellRenderer* p0,GtkCellEditable* p1,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,CellRenderer*,CellEditable* > SlotType;

  auto obj = dynamic_cast<CellArea*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0)
, dynamic_cast<CellEditable*>(Glib::wrap_auto((GObject*)(p1), false))
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo CellArea_signal_remove_editable_info =
{
  "remove-editable",
  (GCallback) &CellArea_signal_remove_editable_callback,
  (GCallback) &CellArea_signal_remove_editable_callback
};


static void CellArea_signal_focus_changed_callback(GtkCellArea* self, GtkCellRenderer* p0,const gchar* p1,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,CellRenderer*,const Glib::ustring& > SlotType;

  auto obj = dynamic_cast<CellArea*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(Glib::wrap(p0)
, Glib::convert_const_gchar_ptr_to_ustring(p1)
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo CellArea_signal_focus_changed_info =
{
  "focus-changed",
  (GCallback) &CellArea_signal_focus_changed_callback,
  (GCallback) &CellArea_signal_focus_changed_callback
};


} // anonymous namespace


namespace Glib
{

Glib::RefPtr<Gtk::CellArea> wrap(GtkCellArea* object, bool take_copy)
{
  return Glib::RefPtr<Gtk::CellArea>( dynamic_cast<Gtk::CellArea*> (Glib::wrap_auto ((GObject*)(object), take_copy)) );
  //We use dynamic_cast<> in case of multiple inheritance.
}

} /* namespace Glib */


namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& CellArea_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &CellArea_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_cell_area_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:
  Buildable::add_interface(get_type());
  CellLayout::add_interface(get_type());

  }

  return *this;
}


void CellArea_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);

  klass->get_request_mode = &get_request_mode_vfunc_callback;
  klass->get_preferred_width = &get_preferred_width_vfunc_callback;
  klass->get_preferred_height_for_width = &get_preferred_height_for_width_vfunc_callback;
  klass->get_preferred_height = &get_preferred_height_vfunc_callback;
  klass->get_preferred_width_for_height = &get_preferred_width_for_height_vfunc_callback;

}

GtkSizeRequestMode CellArea_Class::get_request_mode_vfunc_callback(GtkCellArea* self)
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
        return ((GtkSizeRequestMode)(obj->get_request_mode_vfunc()));
      }
      catch(...)
      {
        Glib::exception_handlers_invoke();
      }
    }
  }

  BaseClassType *const base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(self)) // Get the parent class of the object class (The original underlying C class).
  );

  // Call the original underlying C function:
  if(base && base->get_request_mode)
  {
    GtkSizeRequestMode retval = (*base->get_request_mode)(self);
    return retval;
  }

  typedef GtkSizeRequestMode RType;
  return RType();
}


Glib::ObjectBase* CellArea_Class::wrap_new(GObject* object)
{
  return new CellArea((GtkCellArea*)object);
}


/* The implementation: */

GtkCellArea* CellArea::gobj_copy()
{
  reference();
  return gobj();
}

CellArea::CellArea(const Glib::ConstructParams& construct_params)
:
  Glib::Object(construct_params)
{
   if(gobject_ && g_object_is_floating (gobject_))
     g_object_ref_sink(gobject_); //Stops it from being floating.

}

CellArea::CellArea(GtkCellArea* castitem)
:
  Glib::Object((GObject*)(castitem))
{}


CellArea::CellArea(CellArea&& src) noexcept
: Glib::Object(std::move(src))
  , Buildable(std::move(src))
  , CellLayout(std::move(src))
{}

CellArea& CellArea::operator=(CellArea&& src) noexcept
{
  Glib::Object::operator=(std::move(src));
  Buildable::operator=(std::move(src));
  CellLayout::operator=(std::move(src));
  return *this;
}

CellArea::~CellArea() noexcept
{}


CellArea::CppClassType CellArea::cellarea_class_; // initialize static member

GType CellArea::get_type()
{
  return cellarea_class_.init().get_type();
}


GType CellArea::get_base_type()
{
  return gtk_cell_area_get_type();
}


CellArea::CellArea()
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Glib::Object(Glib::ConstructParams(cellarea_class_.init()))
{
  
   if(gobject_ && g_object_is_floating (gobject_))
     g_object_ref_sink(gobject_); //Stops it from being floating.

}

void CellArea::add(CellRenderer& renderer)
{
  gtk_cell_area_add(gobj(), (renderer).gobj());
}

void CellArea::remove(CellRenderer& renderer)
{
  gtk_cell_area_remove(gobj(), (renderer).gobj());
}

bool CellArea::has_renderer(CellRenderer& renderer)
{
  return gtk_cell_area_has_renderer(gobj(), (renderer).gobj());
}

int CellArea::event(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, GdkEvent* gdk_event, const Gdk::Rectangle& cell_area, GtkCellRendererState flags)
{
  return gtk_cell_area_event(gobj(), Glib::unwrap(context), (widget).gobj(), gdk_event, (cell_area).gobj(), flags);
}

void CellArea::render(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, const ::Cairo::RefPtr< ::Cairo::Context>& cr, const Gdk::Rectangle& background_area, const Gdk::Rectangle& cell_area, CellRendererState flags, bool paint_focus)
{
  gtk_cell_area_render(gobj(), Glib::unwrap(context), (widget).gobj(), (cr)->cobj(), (background_area).gobj(), (cell_area).gobj(), ((GtkCellRendererState)(flags)), static_cast<int>(paint_focus));
}

void CellArea::get_cell_allocation(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, CellRenderer& renderer, const Gdk::Rectangle& cell_area, Gdk::Rectangle& allocation)
{
  gtk_cell_area_get_cell_allocation(gobj(), Glib::unwrap(context), (widget).gobj(), (renderer).gobj(), (cell_area).gobj(), (allocation).gobj());
}

CellRenderer* CellArea::get_cell_at_position(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, const Gdk::Rectangle& cell_area, int x, int y, Gdk::Rectangle& alloc_area)
{
  return Glib::wrap(gtk_cell_area_get_cell_at_position(gobj(), Glib::unwrap(context), (widget).gobj(), (cell_area).gobj(), x, y, (alloc_area).gobj()));
}

const CellRenderer* CellArea::get_cell_at_position(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, const Gdk::Rectangle& cell_area, int x, int y, Gdk::Rectangle& alloc_area) const
{
  return const_cast<CellArea*>(this)->get_cell_at_position(context, widget, cell_area, x, y, alloc_area);
}

Glib::RefPtr<CellAreaContext> CellArea::create_context() const
{
  return Glib::wrap(gtk_cell_area_create_context(const_cast<GtkCellArea*>(gobj())));
}

Glib::RefPtr<CellAreaContext> CellArea::copy_context(const Glib::RefPtr<const CellAreaContext>& context)
{
  return Glib::wrap(gtk_cell_area_copy_context(gobj(), const_cast<GtkCellAreaContext*>(Glib::unwrap(context))));
}

SizeRequestMode CellArea::get_request_mode() const
{
  return ((SizeRequestMode)(gtk_cell_area_get_request_mode(const_cast<GtkCellArea*>(gobj()))));
}

void CellArea::get_preferred_width(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int& minimum_width, int& natural_width)
{
  gtk_cell_area_get_preferred_width(gobj(), Glib::unwrap(context), (widget).gobj(), &(minimum_width), &(natural_width));
}

void CellArea::get_preferred_height_for_width(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int width, int& minimum_height, int& natural_height)
{
  gtk_cell_area_get_preferred_height_for_width(gobj(), Glib::unwrap(context), (widget).gobj(), width, &(minimum_height), &(natural_height));
}

void CellArea::get_preferred_height(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int& minimum_height, int& natural_height)
{
  gtk_cell_area_get_preferred_height(gobj(), Glib::unwrap(context), (widget).gobj(), &(minimum_height), &(natural_height));
}

void CellArea::get_preferred_width_for_height(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int height, int& minimum_width, int& natural_width)
{
  gtk_cell_area_get_preferred_width_for_height(gobj(), Glib::unwrap(context), (widget).gobj(), height, &(minimum_width), &(natural_width));
}

Glib::ustring CellArea::get_current_path_string() const
{
  return Glib::convert_const_gchar_ptr_to_ustring(gtk_cell_area_get_current_path_string(const_cast<GtkCellArea*>(gobj())));
}

void CellArea::apply_attributes(const Glib::RefPtr<TreeModel>& tree_model, const TreeModel::iterator& iter, bool is_expander, bool is_expanded)
{
  gtk_cell_area_apply_attributes(gobj(), Glib::unwrap(tree_model), const_cast<GtkTreeIter*>((iter).gobj()), static_cast<int>(is_expander), static_cast<int>(is_expanded));
}

void CellArea::attribute_connect(CellRenderer& renderer, const Glib::ustring& attribute, int column)
{
  gtk_cell_area_attribute_connect(gobj(), (renderer).gobj(), attribute.c_str(), column);
}

void CellArea::attribute_disconnect(CellRenderer& renderer, const Glib::ustring& attribute)
{
  gtk_cell_area_attribute_disconnect(gobj(), (renderer).gobj(), attribute.c_str());
}

int CellArea::attribute_get_column(CellRenderer& renderer, const Glib::ustring& attribute) const
{
  return gtk_cell_area_attribute_get_column(const_cast<GtkCellArea*>(gobj()), (renderer).gobj(), attribute.c_str());
}

void CellArea::cell_set_property(CellRenderer& renderer, const Glib::ustring& property_name, const Glib::ValueBase& value)
{
  gtk_cell_area_cell_set_property(gobj(), (renderer).gobj(), property_name.c_str(), (value).gobj());
}

void CellArea::cell_get_property(CellRenderer& renderer, const Glib::ustring& property_name, Glib::ValueBase& value)
{
  gtk_cell_area_cell_get_property(gobj(), (renderer).gobj(), property_name.c_str(), (value).gobj());
}

bool CellArea::is_activatable() const
{
  return gtk_cell_area_is_activatable(const_cast<GtkCellArea*>(gobj()));
}

bool CellArea::activate(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, const Gdk::Rectangle& cell_area, CellRendererState flags, bool edit_only)
{
  return gtk_cell_area_activate(gobj(), Glib::unwrap(context), (widget).gobj(), (cell_area).gobj(), ((GtkCellRendererState)(flags)), static_cast<int>(edit_only));
}

bool CellArea::focus(DirectionType direction)
{
  return gtk_cell_area_focus(gobj(), ((GtkDirectionType)(direction)));
}

void CellArea::set_focus_cell(CellRenderer& renderer)
{
  gtk_cell_area_set_focus_cell(gobj(), (renderer).gobj());
}

CellRenderer* CellArea::get_focus_cell()
{
  return Glib::wrap(gtk_cell_area_get_focus_cell(gobj()));
}

const CellRenderer* CellArea::get_focus_cell() const
{
  return const_cast<CellArea*>(this)->get_focus_cell();
}

void CellArea::add_focus_sibling(CellRenderer& renderer, CellRenderer& sibling)
{
  gtk_cell_area_add_focus_sibling(gobj(), (renderer).gobj(), (sibling).gobj());
}

void CellArea::remove_focus_sibling(CellRenderer& renderer, CellRenderer& sibling)
{
  gtk_cell_area_remove_focus_sibling(gobj(), (renderer).gobj(), (sibling).gobj());
}

bool CellArea::is_focus_sibling(CellRenderer& renderer, CellRenderer& sibling)
{
  return gtk_cell_area_is_focus_sibling(gobj(), (renderer).gobj(), (sibling).gobj());
}

std::vector<CellRenderer*> CellArea::get_focus_siblings(CellRenderer& renderer)
{
  return Glib::ListHandler<CellRenderer*>::list_to_vector(const_cast<GList*>(gtk_cell_area_get_focus_siblings(gobj(), (renderer).gobj())), Glib::OWNERSHIP_NONE);
}

std::vector<const CellRenderer*> CellArea::get_focus_siblings(const CellRenderer& renderer) const
{
  return Glib::ListHandler<const CellRenderer*>::list_to_vector(const_cast<GList*>(gtk_cell_area_get_focus_siblings(const_cast<GtkCellArea*>(gobj()), const_cast<GtkCellRenderer*>((renderer).gobj()))), Glib::OWNERSHIP_NONE);
}

CellRenderer* CellArea::get_focus_from_sibling(CellRenderer& renderer)
{
  return Glib::wrap(gtk_cell_area_get_focus_from_sibling(gobj(), (renderer).gobj()));
}

const CellRenderer* CellArea::get_focus_from_sibling(CellRenderer& renderer) const
{
  return const_cast<CellArea*>(this)->get_focus_from_sibling(renderer);
}

CellRenderer* CellArea::get_edited_cell()
{
  return Glib::wrap(gtk_cell_area_get_edited_cell(gobj()));
}

const CellRenderer* CellArea::get_edited_cell() const
{
  return const_cast<CellArea*>(this)->get_edited_cell();
}

CellEditable* CellArea::get_edit_widget()
{
  return dynamic_cast<CellEditable*>(Glib::wrap_auto((GObject*)(gtk_cell_area_get_edit_widget(gobj())), false));
}

const CellEditable* CellArea::get_edit_widget() const
{
  return const_cast<CellArea*>(this)->get_edit_widget();
}

bool CellArea::activate_cell(Widget& widget, CellRenderer& renderer, GdkEvent* gdk_event, const Gdk::Rectangle& cell_area, CellRendererState flags)
{
  return gtk_cell_area_activate_cell(gobj(), (widget).gobj(), (renderer).gobj(), gdk_event, (cell_area).gobj(), ((GtkCellRendererState)(flags)));
}

void CellArea::stop_editing(bool canceled)
{
  gtk_cell_area_stop_editing(gobj(), static_cast<int>(canceled));
}

void CellArea::inner_cell_area(Widget& widget, const Gdk::Rectangle& cell_area, Gdk::Rectangle& inner_area)
{
  gtk_cell_area_inner_cell_area(gobj(), (widget).gobj(), (cell_area).gobj(), (inner_area).gobj());
}

void CellArea::request_renderer(CellRenderer& renderer, Orientation orientation, Widget& widget, int for_size, int& minimum_size, int& natural_size)
{
  gtk_cell_area_request_renderer(gobj(), (renderer).gobj(), ((GtkOrientation)(orientation)), (widget).gobj(), for_size, &(minimum_size), &(natural_size));
}


Glib::SignalProxy4< void,const Glib::RefPtr<TreeModel>&,const TreeModel::iterator&,bool,bool > CellArea::signal_apply_attributes()
{
  return Glib::SignalProxy4< void,const Glib::RefPtr<TreeModel>&,const TreeModel::iterator&,bool,bool >(this, &CellArea_signal_apply_attributes_info);
}


Glib::SignalProxy4< void,CellRenderer*,CellEditable*,const Gdk::Rectangle&,const Glib::ustring& > CellArea::signal_add_editable()
{
  return Glib::SignalProxy4< void,CellRenderer*,CellEditable*,const Gdk::Rectangle&,const Glib::ustring& >(this, &CellArea_signal_add_editable_info);
}


Glib::SignalProxy2< void,CellRenderer*,CellEditable* > CellArea::signal_remove_editable()
{
  return Glib::SignalProxy2< void,CellRenderer*,CellEditable* >(this, &CellArea_signal_remove_editable_info);
}


Glib::SignalProxy2< void,CellRenderer*,const Glib::ustring& > CellArea::signal_focus_changed()
{
  return Glib::SignalProxy2< void,CellRenderer*,const Glib::ustring& >(this, &CellArea_signal_focus_changed_info);
}


Glib::PropertyProxy< CellRenderer* > CellArea::property_focus_cell() 
{
  return Glib::PropertyProxy< CellRenderer* >(this, "focus-cell");
}

Glib::PropertyProxy_ReadOnly< CellRenderer* > CellArea::property_focus_cell() const
{
  return Glib::PropertyProxy_ReadOnly< CellRenderer* >(this, "focus-cell");
}

Glib::PropertyProxy_ReadOnly< CellRenderer* > CellArea::property_edited_cell() const
{
  return Glib::PropertyProxy_ReadOnly< CellRenderer* >(this, "edited-cell");
}

Glib::PropertyProxy_ReadOnly< CellEditable* > CellArea::property_edit_widget() const
{
  return Glib::PropertyProxy_ReadOnly< CellEditable* >(this, "edit-widget");
}


SizeRequestMode Gtk::CellArea::get_request_mode_vfunc() const
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->get_request_mode)
  {
    SizeRequestMode retval(((SizeRequestMode)((*base->get_request_mode)(const_cast<GtkCellArea*>(gobj())))));
    return retval;
  }

  typedef SizeRequestMode RType;
  return RType();
}
void Gtk::CellArea::get_preferred_width_vfunc(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int& minimum_width, int& natural_width) 
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->get_preferred_width)
  {
    (*base->get_preferred_width)(gobj(),Glib::unwrap(context),(widget).gobj(),&(minimum_width),&(natural_width));
  }
}
void Gtk::CellArea::get_preferred_height_for_width_vfunc(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int width, int& minimum_height, int& natural_height) 
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->get_preferred_height_for_width)
  {
    (*base->get_preferred_height_for_width)(gobj(),Glib::unwrap(context),(widget).gobj(),width,&(minimum_height),&(natural_height));
  }
}
void Gtk::CellArea::get_preferred_height_vfunc(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int& minimum_height, int& natural_height) 
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->get_preferred_height)
  {
    (*base->get_preferred_height)(gobj(),Glib::unwrap(context),(widget).gobj(),&(minimum_height),&(natural_height));
  }
}
void Gtk::CellArea::get_preferred_width_for_height_vfunc(const Glib::RefPtr<CellAreaContext>& context, Widget& widget, int height, int& minimum_width, int& natural_width) 
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->get_preferred_width_for_height)
  {
    (*base->get_preferred_width_for_height)(gobj(),Glib::unwrap(context),(widget).gobj(),height,&(minimum_width),&(natural_width));
  }
}


} // namespace Gtk

